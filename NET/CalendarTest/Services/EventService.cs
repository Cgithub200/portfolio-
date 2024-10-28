using Azure.Core;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;
using Microsoft.IdentityModel.Tokens;
using System.Diagnostics;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using VertoTest.Context;
using VertoTest.Models;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace VertoTest.Services
{
    public class EventService : EventInterface
    {
        private readonly CalendarContext context;
        private readonly IMemoryCache cache;
        public EventService(CalendarContext context, IMemoryCache cache)
        {
            this.context = context;
            this.cache = cache;
        }

        // Responsible for deleating existing table entries
        public async Task<bool> Delete(CalendarEvent content)
        {
            this.context.Events.Remove(content);
            await this.context.SaveChangesAsync();
            WipeCache(content.EventStart, content.EventEnd);
            return true;
        }

        // Handles event edits
        public async Task<bool> Edit(CalendarEvent existingEvent, EditEventRequest request)
        {
            if (request.EventData.EventStart >= request.EventData.EventEnd)
            {
                return false;
            }

            // Cache must be wiped between the time period the edit will effect
            DateTime minDate = DateTime.Compare(existingEvent.EventStart, request.EventData.EventStart) < 0 ? existingEvent.EventStart : request.EventData.EventStart;
            DateTime maxDate = DateTime.Compare(existingEvent.EventEnd, request.EventData.EventEnd) > 0 ? existingEvent.EventEnd : request.EventData.EventEnd;
            WipeCache(minDate, maxDate);

            // Updates the event to use the new details
            existingEvent.Title = request.EventData.Title;
            existingEvent.Description = request.EventData.Description;

            existingEvent.EventStart = request.EventData.EventStart;
            existingEvent.EventEnd = request.EventData.EventEnd;
            existingEvent.LastUpdated = DateTime.UtcNow;

            await this.context.SaveChangesAsync();
            return true;
        }

        // Returns a sorted list of all events within a day
        public async Task<List<CalendarEvent>> List(DateTime lowerRange)
        {
            DateTime upperRange = lowerRange.AddDays(1);
            return await ListRange(lowerRange, upperRange);
        }

        // Returns all events within a set range
        public async Task<List<CalendarEvent>> ListRange(DateTime lowerRange, DateTime upperRange)
        {
            var cacheKey = $"event{lowerRange}_{upperRange}";
            if (!cache.TryGetValue(cacheKey, out List<CalendarEvent> events))
            {
                events = await context.Events.Where(e => e.EventStart < upperRange && e.EventEnd > lowerRange) // Returns all events within the range
                .OrderBy(e => e.EventStart) // Ordered to ensure overlapping events are still clickable
                .ToListAsync();

                // Adds to the cache to reduce future lookups
                var cacheOptions = new MemoryCacheEntryOptions()
                .SetSlidingExpiration(TimeSpan.FromMinutes(30)) 
                .SetAbsoluteExpiration(TimeSpan.FromHours(1));
                cache.Set(cacheKey, events, cacheOptions);

            } 
            return events;
        }

        // This checks to see if there are any events within a range usualy a day
        public async Task<List<int>> CheckDaysWithEventsInRange(DateTime lowerRange, DateTime upperRange)
        {
            var dayCounter = 1;
            var daysWithEvents = new List<int>();
            // Fills the list of which days have events
            for (var day = lowerRange; day <= upperRange; day = day.AddDays(1))
            {
                var events = await List(day);
                if (events.Any())
                {
                    daysWithEvents.Add(dayCounter);
                } 
                dayCounter += 1;
            }
            return daysWithEvents;
        }

        // Adds a new element to the database
        public async Task<bool> Add(CalendarEvent content)
        {
            if (content.EventStart >= content.EventEnd)
            {
                return false;
            }
            await this.context.Events.AddAsync(content);
            await this.context.SaveChangesAsync();
            WipeCache(content.EventStart, content.EventEnd);
            return true;
        }

        // Returns the item with a matching primary key
        public async Task<CalendarEvent?> GetViaId(int id)
        {
            return await this.context.Events.FirstOrDefaultAsync(myevent => myevent.EventID == id);
        }

        // Checks to see if a event already exists in table
        public async Task<bool> Exists(DateTime lowerRange)
        {
            DateTime upperRange = lowerRange.AddDays(1);
            var eventsExist = await context.Events.Where(e => e.EventStart < upperRange && e.EventEnd > lowerRange).AnyAsync();
            return eventsExist;
        }

        // This function determins the validity of the token passed and returns its event
        public async Task<(CalendarEvent Event, string ErrorMessage)> ValidateAndRetrieveEventByToken(string token, byte[] key)
        {
            var tokenHandler = new JwtSecurityTokenHandler();
            ClaimsPrincipal principal;

            // Validates the token
            try
            {
                principal = tokenHandler.ValidateToken(token, new TokenValidationParameters
                {
                    ValidateIssuerSigningKey = true,
                    IssuerSigningKey = new SymmetricSecurityKey(key),
                    ValidateIssuer = false,
                    ValidateAudience = false
                }, out SecurityToken validatedToken);
            }
            catch
            {
                return (null, "Incorrect token");
            }

            // Extracts the event ID from the token
            var tokenId = principal.FindFirst("id");

            if (tokenId == null || !int.TryParse(tokenId.Value, out int eventId))
            {
                return (null, "Token data issue");
            }

            // Retrieves the event by ID
            var existingEvent = await GetViaId(eventId);
            if (existingEvent == null)
            {
                return (null, "Cannot find event");
            }

            return (existingEvent, null); // Return the event if successful
        }




        // Creates a new token primarly to hide sensetive data from the user
        string EventInterface.GenerateToken(JwtSecurityTokenHandler tokenHandler, byte[] key, int eventId)
        {
            var tokenDescriptor = new SecurityTokenDescriptor
            {
                Subject = new ClaimsIdentity(new Claim[]
            {
            new Claim("id", eventId.ToString())
                }),
                Expires = DateTime.UtcNow.AddHours(1), 
                SigningCredentials = new SigningCredentials(new SymmetricSecurityKey(key), SecurityAlgorithms.HmacSha256Signature)
            };

            var token = tokenHandler.CreateToken(tokenDescriptor);
            return tokenHandler.WriteToken(token);
        }

        // This function is responsible to clearning all cache records within a date range
        private void WipeCache(DateTime lowerRange, DateTime upperRange)
        {
        // Gets the date without regard to the time
        DateTime startDate = lowerRange.Date;
        DateTime endDate = upperRange.Date;
            Debug.WriteLine($"my startDate: {startDate}");

        // Iterativly wipes the cache day by day for the span of the event
        for (DateTime current = startDate; current <= endDate; current = current.AddDays(1))
        {
            DateTime previousDayEndTime = current.AddDays(-1).Date.AddHours(23);
            DateTime nextDayStartTime = current.Date.AddHours(23);
            string key = $"event{previousDayEndTime:dd/MM/yyyy HH:mm:ss}_{nextDayStartTime:dd/MM/yyyy HH:mm:ss}";

            cache.Remove(key);
            Debug.WriteLine($"my2 startDate: {current}");
        }  
        }
    }
}
